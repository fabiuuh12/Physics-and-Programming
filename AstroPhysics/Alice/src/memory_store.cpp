#include "alice/memory_store.hpp"

#include "alice/string_utils.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace alice {

MemoryStore::MemoryStore(std::filesystem::path db_path) : db_path_(std::move(db_path)) {
    db_path_ = std::filesystem::weakly_canonical(db_path_.is_absolute() ? db_path_ : std::filesystem::current_path() / db_path_);
    std::filesystem::create_directories(db_path_.parent_path());
    load();
}

const std::filesystem::path& MemoryStore::db_path() const {
    return db_path_;
}

int MemoryStore::count() const {
    return static_cast<int>(items_.size());
}

std::string MemoryStore::escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string MemoryStore::unescape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '\\' || i + 1 >= value.size()) {
            out.push_back(value[i]);
            continue;
        }
        const char next = value[i + 1];
        if (next == 't') {
            out.push_back('\t');
            ++i;
        } else if (next == 'n') {
            out.push_back('\n');
            ++i;
        } else {
            out.push_back(next);
            ++i;
        }
    }
    return out;
}

void MemoryStore::load() {
    items_.clear();
    next_id_ = 1;

    std::ifstream file(db_path_);
    if (!file.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (trim(line).empty()) {
            continue;
        }

        std::vector<std::string> parts;
        std::string piece;
        std::istringstream row(line);
        while (std::getline(row, piece, '\t')) {
            parts.push_back(piece);
        }
        if (parts.size() < 5) {
            continue;
        }

        MemoryItem item;
        try {
            item.id = std::stoi(parts[0]);
            item.use_count = std::stoi(parts[1]);
        } catch (...) {
            continue;
        }
        item.created_at = unescape(parts[2]);
        item.category = unescape(parts[3]);
        item.content = unescape(parts[4]);

        items_.push_back(item);
        next_id_ = std::max(next_id_, item.id + 1);
    }
}

bool MemoryStore::save() const {
    std::ofstream file(db_path_, std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& item : items_) {
        file << item.id << "\t" << item.use_count << "\t" << escape(item.created_at) << "\t" << escape(item.category)
             << "\t" << escape(item.content) << "\n";
    }
    return file.good();
}

bool MemoryStore::add(const std::string& content, const std::string& category) {
    const std::string cleaned = trim(content);
    if (cleaned.empty()) {
        return false;
    }
    const std::string normalized_new = normalize_text(cleaned);
    if (normalized_new.empty()) {
        return false;
    }

    for (const auto& item : items_) {
        if (normalize_text(item.content) == normalized_new) {
            return false;
        }
    }

    MemoryItem item;
    item.id = next_id_++;
    item.content = cleaned;
    item.category = category;
    item.created_at = now_iso8601();
    item.use_count = 0;

    items_.push_back(item);
    return save();
}

std::vector<MemoryItem> MemoryStore::recent(int limit) {
    if (limit <= 0) {
        limit = 5;
    }
    std::vector<MemoryItem> out;
    const int start = std::max(0, static_cast<int>(items_.size()) - limit);
    for (int i = static_cast<int>(items_.size()) - 1; i >= start; --i) {
        out.push_back(items_[static_cast<std::size_t>(i)]);
    }
    return out;
}

std::vector<MemoryItem> MemoryStore::search(const std::string& query, int limit) {
    if (limit <= 0) {
        limit = 5;
    }
    const std::string q = normalize_text(query);
    if (q.empty()) {
        return recent(limit);
    }
    const auto q_tokens = split_words(q);

    std::vector<std::pair<int, std::size_t>> scored;
    scored.reserve(items_.size());

    for (std::size_t i = 0; i < items_.size(); ++i) {
        const std::string normalized = normalize_text(items_[i].content);
        int score = 0;
        if (normalized.find(q) != std::string::npos) {
            score += 8;
        }
        for (const auto& token : q_tokens) {
            if (token.size() < 3) {
                continue;
            }
            if (normalized.find(token) != std::string::npos) {
                score += 2;
            }
        }
        score += std::min(items_[i].use_count, 5);
        if (score > 0) {
            scored.emplace_back(score, i);
        }
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    std::vector<MemoryItem> result;
    for (int i = 0; i < static_cast<int>(scored.size()) && i < limit; ++i) {
        auto& item = items_[scored[static_cast<std::size_t>(i)].second];
        item.use_count += 1;
        result.push_back(item);
    }

    if (!result.empty()) {
        save();
    }
    return result;
}

}  // namespace alice
