#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t RECORD_SIZE = 32;
constexpr std::uint64_t DEFAULT_BUFFER_MIB = 64;
constexpr double DEFAULT_TARGET_GIB = 5.0;

#pragma pack(push, 1)
struct BulletRecord {
    std::uint64_t occupancy;
    std::uint8_t pieces[16];
    std::int16_t score;
    std::uint8_t result;
    std::uint8_t king_square;
    std::uint8_t opponent_king_square;
    std::uint8_t extra[3];
};

struct TunguskaRecord {
    std::uint64_t occupancy;
    std::uint8_t pieces[16];
    std::int16_t score;
    std::uint8_t result;
    std::uint8_t stm;
    std::uint16_t pad1;
    std::uint16_t pad2;
};
#pragma pack(pop)

static_assert(sizeof(BulletRecord) == RECORD_SIZE, "Bullet record must be 32 bytes");
static_assert(sizeof(TunguskaRecord) == RECORD_SIZE, "Tunguska record must be 32 bytes");

struct Options {
    fs::path input;
    fs::path output_prefix;
    double target_size_gib = DEFAULT_TARGET_GIB;
    std::uint64_t buffer_mib = DEFAULT_BUFFER_MIB;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void print_usage(const char* executable) {
    std::cerr << "Usage:\n  " << executable
              << " bullet-to-tunguska <input.bin> <output-prefix>"
                 " [--target-size-gib N] [--buffer-mib N]\n";
}

double parse_positive_double(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        fail(std::string("Invalid value for ") + option + ": " + text);
    }
    if (consumed != text.size() || !std::isfinite(value) || value <= 0.0) {
        fail(std::string("Invalid value for ") + option + ": " + text);
    }
    return value;
}

std::uint64_t parse_positive_integer(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        fail(std::string("Invalid value for ") + option + ": " + text);
    }
    if (consumed != text.size() || value == 0) {
        fail(std::string("Invalid value for ") + option + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

Options parse_options(int argc, char* argv[]) {
    if (argc < 4 || std::string(argv[1]) != "bullet-to-tunguska") {
        print_usage(argv[0]);
        fail("Invalid command line");
    }

    Options options{argv[2], argv[3]};
    for (int i = 4; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc) {
            fail("Missing value after " + option);
        }
        if (option == "--target-size-gib") {
            options.target_size_gib = parse_positive_double(argv[++i], "--target-size-gib");
        } else if (option == "--buffer-mib") {
            options.buffer_mib = parse_positive_integer(argv[++i], "--buffer-mib");
        } else {
            fail("Unknown option: " + option);
        }
    }
    return options;
}

unsigned trailing_zeroes(std::uint64_t value) {
#ifdef _MSC_VER
    unsigned long index = 0;
#ifdef _WIN64
    _BitScanForward64(&index, value);
#else
    const auto low = static_cast<unsigned long>(value);
    if (low != 0) {
        _BitScanForward(&index, low);
    } else {
        _BitScanForward(&index, static_cast<unsigned long>(value >> 32));
        index += 32;
    }
#endif
    return static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_ctzll(value));
#endif
}

fs::path shard_path(const fs::path& prefix, std::uint64_t index) {
    std::ostringstream suffix;
    suffix << "_chunk_" << std::setw(3) << std::setfill('0') << index << ".bin";
    fs::path result = prefix;
    result += suffix.str();
    return result;
}

TunguskaRecord convert_record(const BulletRecord& input, std::uint64_t record_index) {
    if (input.result > 2) {
        fail("Invalid result at record " + std::to_string(record_index));
    }
    if (input.king_square >= 64 || input.opponent_king_square >= 64) {
        fail("Invalid king-square metadata at record " + std::to_string(record_index));
    }

    TunguskaRecord output{};
    output.occupancy = input.occupancy;
    output.score = input.score;
    output.result = input.result;
    output.stm = 0;

    std::uint64_t occupancy = input.occupancy;
    unsigned piece_index = 0;
    int own_king_square = -1;
    int opponent_king_square = -1;

    while (occupancy != 0) {
        if (piece_index >= 32) {
            fail("More than 32 occupied squares at record " + std::to_string(record_index));
        }
        const unsigned square = trailing_zeroes(occupancy);
        const std::uint8_t packed = input.pieces[piece_index / 2];
        const std::uint8_t bullet_piece =
            static_cast<std::uint8_t>((packed >> (4 * (piece_index & 1))) & 0x0F);
        if (!((bullet_piece <= 5) || (bullet_piece >= 8 && bullet_piece <= 13))) {
            fail("Invalid Bullet piece code at record " + std::to_string(record_index));
        }

        const std::uint8_t tunguska_piece = bullet_piece >= 8
            ? static_cast<std::uint8_t>(bullet_piece - 2)
            : bullet_piece;
        output.pieces[piece_index / 2] |=
            static_cast<std::uint8_t>(tunguska_piece << (4 * (piece_index & 1)));

        if (bullet_piece == 5) {
            if (own_king_square != -1) {
                fail("Multiple current-player kings at record " + std::to_string(record_index));
            }
            own_king_square = static_cast<int>(square);
        } else if (bullet_piece == 13) {
            if (opponent_king_square != -1) {
                fail("Multiple opponent kings at record " + std::to_string(record_index));
            }
            opponent_king_square = static_cast<int>(square);
        }

        occupancy &= occupancy - 1;
        ++piece_index;
    }

    if (own_king_square != input.king_square) {
        fail("Current-player king does not match metadata at record " +
             std::to_string(record_index));
    }
    if ((opponent_king_square ^ 56) != input.opponent_king_square) {
        fail("Opponent king does not match metadata at record " +
             std::to_string(record_index));
    }
    return output;
}

void check_output_paths(const Options& options, std::uint64_t shard_count) {
    for (std::uint64_t i = 0; i < shard_count; ++i) {
        const fs::path final_path = shard_path(options.output_prefix, i);
        fs::path partial_path = final_path;
        partial_path += ".part";
        if (fs::exists(final_path) || fs::exists(partial_path)) {
            fail("Refusing to overwrite existing output: " + final_path.string());
        }
    }
}

void print_progress(std::uint64_t converted, std::uint64_t total,
                    std::chrono::steady_clock::time_point started) {
    using seconds = std::chrono::duration<double>;
    const double elapsed = std::max(seconds(std::chrono::steady_clock::now() - started).count(), 0.001);
    const double fraction = static_cast<double>(converted) / static_cast<double>(total);
    const double mib_per_second = (static_cast<double>(converted) * RECORD_SIZE) /
                                  (1024.0 * 1024.0 * elapsed);
    const double eta = fraction > 0.0 ? elapsed * (1.0 - fraction) / fraction : 0.0;
    std::cout << '\r' << std::fixed << std::setprecision(2)
              << std::setw(6) << (100.0 * fraction) << "%  "
              << converted << '/' << total << " records  "
              << mib_per_second << " MiB/s  ETA "
              << static_cast<std::uint64_t>(eta) << " s" << std::flush;
}

void convert(const Options& options) {
    if (!fs::is_regular_file(options.input)) {
        fail("Input is not a regular file: " + options.input.string());
    }
    const std::uint64_t input_size = fs::file_size(options.input);
    if (input_size == 0 || input_size % RECORD_SIZE != 0) {
        fail("Input size must be a nonzero multiple of 32 bytes");
    }
    const std::uint64_t total_records = input_size / RECORD_SIZE;

    const long double target_bytes = static_cast<long double>(options.target_size_gib) *
                                     1024.0L * 1024.0L * 1024.0L;
    const long double ideal_shards = static_cast<long double>(input_size) / target_bytes;
    if (ideal_shards > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        fail("Requested target size produces too many shards");
    }
    const std::uint64_t shard_count = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(std::llround(ideal_shards)));

    fs::path output_directory = options.output_prefix.parent_path();
    if (output_directory.empty()) {
        output_directory = fs::current_path();
    } else {
        fs::create_directories(output_directory);
    }
    check_output_paths(options, shard_count);

    std::error_code space_error;
    const fs::space_info space = fs::space(output_directory, space_error);
    if (!space_error && space.available < input_size) {
        fail("Insufficient free space: conversion requires at least " +
             std::to_string(input_size) + " available bytes");
    }

    if (options.buffer_mib > std::numeric_limits<std::uint64_t>::max() / (1024ULL * 1024ULL)) {
        fail("--buffer-mib is too large");
    }
    const std::uint64_t buffer_bytes = options.buffer_mib * 1024ULL * 1024ULL;
    const std::uint64_t buffer_records = std::max<std::uint64_t>(1, buffer_bytes / RECORD_SIZE);
    if (buffer_records > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        fail("Requested buffer does not fit in memory");
    }

    std::ifstream input(options.input, std::ios::binary);
    if (!input) {
        fail("Could not open input: " + options.input.string());
    }
    std::vector<BulletRecord> input_buffer(static_cast<std::size_t>(buffer_records));
    std::vector<TunguskaRecord> output_buffer(static_cast<std::size_t>(buffer_records));

    const std::uint64_t base_records = total_records / shard_count;
    const std::uint64_t larger_shards = total_records % shard_count;
    std::uint64_t converted_records = 0;
    const auto started = std::chrono::steady_clock::now();
    auto last_progress = started;

    std::cout << "Input:   " << options.input << '\n'
              << "Records: " << total_records << '\n'
              << "Shards:  " << shard_count << '\n';

    for (std::uint64_t shard = 0; shard < shard_count; ++shard) {
        const std::uint64_t records_in_shard = base_records + (shard < larger_shards ? 1 : 0);
        const fs::path final_path = shard_path(options.output_prefix, shard);
        fs::path partial_path = final_path;
        partial_path += ".part";
        std::ofstream output(partial_path, std::ios::binary);
        if (!output) {
            fail("Could not open output: " + partial_path.string());
        }

        std::uint64_t shard_written = 0;
        while (shard_written < records_in_shard) {
            const std::uint64_t count = std::min(buffer_records, records_in_shard - shard_written);
            const std::streamsize bytes = static_cast<std::streamsize>(count * RECORD_SIZE);
            input.read(reinterpret_cast<char*>(input_buffer.data()), bytes);
            if (input.gcount() != bytes) {
                fail("Unexpected end of input at record " + std::to_string(converted_records));
            }
            for (std::uint64_t i = 0; i < count; ++i) {
                output_buffer[static_cast<std::size_t>(i)] =
                    convert_record(input_buffer[static_cast<std::size_t>(i)], converted_records + i);
            }
            output.write(reinterpret_cast<const char*>(output_buffer.data()), bytes);
            if (!output) {
                fail("Write failed for " + partial_path.string());
            }

            shard_written += count;
            converted_records += count;
            const auto now = std::chrono::steady_clock::now();
            if (now - last_progress >= std::chrono::seconds(1)) {
                print_progress(converted_records, total_records, started);
                last_progress = now;
            }
        }

        output.close();
        if (!output) {
            fail("Failed to finalize " + partial_path.string());
        }
        fs::rename(partial_path, final_path);
        std::cout << "\nCompleted " << final_path << " (" << records_in_shard
                  << " records, " << records_in_shard * RECORD_SIZE << " bytes)\n";
    }

    print_progress(converted_records, total_records, started);
    std::cout << "\nConversion complete.\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        convert(parse_options(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
