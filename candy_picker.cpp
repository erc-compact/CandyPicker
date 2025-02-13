#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <tuple>
#include <tinyxml2.h> // Include tinyxml2 for XML parsing
#include <filesystem>
#include <unordered_set>
#include <iomanip>
#include <getopt.h>
#include <cstdlib>  
namespace fs = std::filesystem;
#define SPEED_OF_LIGHT 299792458.0


// Forward declaration
class Candidate;

/**
 * A custom hash functor for Candidate.
 * This is needed because we're using Candidate inside an std::unordered_set,
 * and the standard library doesn't know how to hash our user-defined type by default.
 */
struct CandidateHash {
    std::size_t operator()(const Candidate& c) const;
};

/**
 * A custom equality functor for Candidate, using Candidate::operator==
 */
struct CandidateEqual {
    bool operator()(const Candidate& a, const Candidate& b) const;
};


std::vector<std::string> get_xml_files(const std::string& root_dir) {
    std::vector<std::string> xml_files;
    try {
        // Recursively iterate over the directory
        for (const auto& entry : fs::recursive_directory_iterator(root_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".xml") {
                xml_files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error accessing files: " << e.what() << std::endl;
    }
    return xml_files;
}


class Candidate {
public:
    double snr;
    double period;
    double f0;
    double dm;
    double acc;
    int nh;
    float ddm_count_ratio;
    float ddm_snr_ratio;
    int nassoc;
    int period_ms;
    double pulse_width;
    std::string search_candidates_database_uuid;
    std::string xml_file_name;
    int candidate_id_in_file;
    bool is_pivot = false;

    // Holds the "related" candidates for clustering
    std::vector<Candidate> related_candidates;

    Candidate()
        : snr(0),
          period(0),
          f0(0),
          dm(0),
          acc(0),
          nh(0),
          ddm_count_ratio(0),
          ddm_snr_ratio(0),
          nassoc(0),
          period_ms(0),
          pulse_width(0),
          search_candidates_database_uuid("") {}

    Candidate(double snr,
              double period,
              double dm,
              double acc,
              int nh,
              float ddm_count_ratio,
              float ddm_snr_ratio,
              int nassoc,
              const std::string& search_candidates_database_uuid,
              const std::string& xml_file_name,
              int candidate_id_in_file)
        : snr(snr),
          period(period),
          dm(dm),
          acc(acc),
          nh(nh),
          ddm_count_ratio(ddm_count_ratio),
          ddm_snr_ratio(ddm_snr_ratio),
          nassoc(nassoc),
          search_candidates_database_uuid(search_candidates_database_uuid),
          xml_file_name(xml_file_name),
          candidate_id_in_file(candidate_id_in_file)
    {
        f0 = 1.0 / period;
        period_ms = static_cast<int>(std::round(period * 1000.0));
        // Example for pulse width; can tweak if needed
        pulse_width = period / std::pow(2.0, nh);
    }

    bool operator<(const Candidate& other) const {
        // Just comparing by period here, but you could change as needed
        return period < other.period;
    }

    bool operator==(const Candidate& other) const {
        // Must match how we define the hash
        // For example, compare these four parameters:
        return (std::fabs(period - other.period) < 1e-12) &&
               (std::fabs(dm - other.dm) < 1e-6) &&
               (std::fabs(acc - other.acc) < 1e-6) &&
               (nh == other.nh);
    }

    bool operator!=(const Candidate& other) const {
        return !(*this == other);
    }
    void make_pivot() {
        is_pivot = true;
    }

    /**
     * Decide if two candidates are "related"
     * period_thresh is the direct threshold in absolute difference,
     * tobs_over_c is (Tobs / speed_of_light) used to correct acceleration effect.
     */
    bool is_related(const Candidate& other, double period_thresh, double tobs_over_c) const {
        // Arbitrary example: DM must be within 100, period must be close after acceleration correction
        const double dm_thresh = 100.0;
        // DM difference check
        if (std::fabs(dm - other.dm) > dm_thresh) return false;

        // Attempt a basic period correction: 
        // corrected_other_period = f0 - (acc_diff) * f0 * (Tobs/c)
        double corrected_other_period = 1/(other.f0 - (other.acc - acc) * other.f0 * tobs_over_c);
       
        double true_period_difference =
            (period / corrected_other_period) > 1.0
                ? std::fmod(period, corrected_other_period)
                : std::fmod(corrected_other_period, period);

        return true_period_difference <= period_thresh || 
               std::fabs(period - corrected_other_period) <= period_thresh;
        // return (std::fabs(period - corrected_other_period) <= period_thresh) ||
        //        (true_period_difference <= period_thresh) ||
        //        (std::fmod(corrected_other_period, period) <= period_thresh);
    }

    /**
     * Add to the list of related candidates
     */
    void add_related(const Candidate& other) {
        related_candidates.push_back(other);
    }


}; // end of Candidate class


// Define the hash for Candidate based on how we define operator==
std::size_t CandidateHash::operator()(const Candidate& c) const {
    // A simple approach: We combine integer-rounded versions of the relevant fields
    // for period, DM, and ACC, plus the int field NH. 
    // The scaling factors (e.g. 1e10 for period) depend on your numeric range.
    // This is simplistic and might cause collisions with real data, but suffices as a demo.

    // Turn period, dm, acc into integers by scaling:
    auto h1 = std::hash<long long>()(static_cast<long long>(std::llround(c.period * 1e10)));
    auto h2 = std::hash<long long>()(static_cast<long long>(std::llround(c.dm     * 1e6)));
    auto h3 = std::hash<long long>()(static_cast<long long>(std::llround(c.acc    * 1e6)));
    auto h4 = std::hash<int>()(c.nh);

    // Combine them with a common technique
    auto combine = [](std::size_t seed, std::size_t v) {
        const std::size_t kMul = 0x9e3779b97f4a7c15ULL;
        seed ^= v + kMul + (seed << 6) + (seed >> 2);
        return seed;
    };

    std::size_t result = h1;
    result = combine(result, h2);
    result = combine(result, h3);
    result = combine(result, h4);

    return result;
}

bool CandidateEqual::operator()(const Candidate& a, const Candidate& b) const {
    return a == b;
}


/**
 * Parse candidate data from an XML file using tinyxml2.
 */
std::vector<Candidate> parse_candidates_from_file(const std::string& xml_file) {
    std::vector<Candidate> candidates;
    try {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(xml_file.c_str()) != tinyxml2::XML_SUCCESS) {
            throw std::runtime_error("Failed to load file: " + xml_file);
        }

        tinyxml2::XMLElement* root = doc.RootElement();
        if (!root) {
            throw std::runtime_error("No root element in XML for " + xml_file);
        }

        tinyxml2::XMLElement* candidatesElem = root->FirstChildElement("candidates");
        if (!candidatesElem) {
            throw std::runtime_error("No <candidates> element found in " + xml_file);
        }

        for (tinyxml2::XMLElement* candidateElem = candidatesElem->FirstChildElement("candidate");
             candidateElem != nullptr;
             candidateElem = candidateElem->NextSiblingElement("candidate"))
        {
            // Safely get the text contents. If any child is missing, this throws an exception.
            const char* periodText                = candidateElem->FirstChildElement("period")->GetText();
            const char* dmText                    = candidateElem->FirstChildElement("dm")->GetText();
            const char* accText                   = candidateElem->FirstChildElement("acc")->GetText();
            const char* nhText                    = candidateElem->FirstChildElement("nh")->GetText();
            const char* snrText                   = candidateElem->FirstChildElement("snr")->GetText();
            const char* ddmCountRatioText         = candidateElem->FirstChildElement("ddm_count_ratio")->GetText();
            const char* ddmSnrRatioText           = candidateElem->FirstChildElement("ddm_snr_ratio")->GetText();
            const char* nassocText                = candidateElem->FirstChildElement("nassoc")->GetText();
            const char* uuidText                  = candidateElem->FirstChildElement("search_candidates_database_uuid")->GetText();
            const char* candidateId              = candidateElem->Attribute("id");

            if (!periodText || !dmText || !accText || !nhText || !snrText ||
                !ddmCountRatioText || !ddmSnrRatioText || !nassocText || !uuidText || !candidateId)
            {
                std::cerr << "xml has incomplete elements, skipping this one: " << xml_file << std::endl;
                continue;
            }

            double period                 = std::stod(periodText);
            double dm                     = std::stod(dmText);
            double acc                    = std::stod(accText);
            int nh                        = std::stoi(nhText);
            double snr                    = std::stod(snrText);
            float ddm_count_ratio         = std::stof(ddmCountRatioText);
            float ddm_snr_ratio           = std::stof(ddmSnrRatioText);
            int nassoc                    = std::stoi(nassocText);
            std::string uuid              = uuidText;
            int candidate_id              = std::stoi(candidateId);

            // Emplace a new Candidate
            candidates.emplace_back(
                snr, period, dm, acc,
                nh, ddm_count_ratio, ddm_snr_ratio,
                nassoc, uuid, xml_file, candidate_id
            );
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing " << xml_file << ": " << e.what() << std::endl;
    }
    return candidates;
}


/**
 * A simple clustering function that iterates over all pairs
 * and checks if they're "related".
 */
void cluster_candidates(std::vector<Candidate>& candidates, double period_thresh) {
    size_t n = candidates.size();

    // For period correction: Tobs = 7200s (2 hours?), tobs_over_c = 7200 / c
    double tobs_over_c = 7200.0 / SPEED_OF_LIGHT;

    // Parallel pairwise check
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (candidates[i].is_related(candidates[j], period_thresh, tobs_over_c)) {
                #pragma omp critical
                {
                    candidates[i].add_related(candidates[j]);
                }
            }
        }
    }
    std::cout << "Finished clustering." << std::endl;
}


/**
 * Helper function that “shortlists” the pivot candidates (removes duplicates or
 * heavily-related ones) by marking everything in related_candidates for removal,
 * then removing them from the main list.
 */
std::vector<Candidate> shortlist_candidates(std::vector<Candidate>& candidates) {
    // We'll store "to_remove" as a set of Candidates (with hashing)
    std::unordered_set<Candidate, CandidateHash, CandidateEqual> to_remove; 
    std::vector<Candidate> shortlisted_candidates;

    // Step 1: Mark candidates to remove if they are in the "related" list of any
    // candidate that has more than 1 related candidate.
    #pragma omp parallel
    {
        // Thread-local sets & vectors
        std::unordered_set<Candidate, CandidateHash, CandidateEqual> local_to_remove;
        std::vector<Candidate> local_shortlisted;

        #pragma omp for nowait
        for (size_t i = 0; i < candidates.size(); ++i) {
            // If a candidate has more than 1 related candidate, remove the related ones
            if (candidates[i].related_candidates.size() > 1) {
                for (const auto& related : candidates[i].related_candidates) {
                    local_to_remove.insert(related);
                }
            }
            // Keep track of this candidate in our local shortlist
            local_shortlisted.push_back(candidates[i]);
        }

        // Merge thread-local results into global
        #pragma omp critical
        {
            to_remove.insert(local_to_remove.begin(), local_to_remove.end());
            shortlisted_candidates.insert(shortlisted_candidates.end(),
                                          local_shortlisted.begin(),
                                          local_shortlisted.end());
        }
    }

    // Step 2: Filter out candidates marked for removal
    // Nullify them first
    #pragma omp parallel for
    for (size_t i = 0; i < shortlisted_candidates.size(); ++i) {
        if (to_remove.find(shortlisted_candidates[i]) != to_remove.end()) {
            shortlisted_candidates[i] = Candidate(); // Default => “null” marker
        }
    }


    // Remove the nullified ones
    shortlisted_candidates.erase(
        std::remove_if(shortlisted_candidates.begin(),
                       shortlisted_candidates.end(),
                       [](const Candidate& c) { return c == Candidate(); }),
        shortlisted_candidates.end());  


    //mark all pivots in original list if they are in shortlisted_candidates
    for (auto& c : candidates) {
        if (std::find(shortlisted_candidates.begin(), shortlisted_candidates.end(), c) != shortlisted_candidates.end()) {
            c.make_pivot();
        }
    }
    

    return shortlisted_candidates;
}

void save_all_candidates_to_csv(const std::vector<Candidate>& candidates, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    // Write header
    ofs << "snr,period,dm,acc,nh,ddm_count_ratio,ddm_snr_ratio,nassoc,"
    << "search_uuid,xml_file,candidate_id_in_file,to_fold\n";

    // Write data rows
    for (const auto& c : candidates) {
        ofs << c.snr << ","
        << std::fixed << std::setprecision(17) << c.period << "," << std::fixed << std::setprecision(8)
        << c.dm << ","
        << c.acc << ","
        << c.nh << ","
        << c.ddm_count_ratio << ","
        << c.ddm_snr_ratio << ","
        << c.nassoc << ","
        << c.period_ms << ","
        << c.search_candidates_database_uuid << ","
        << c.xml_file_name << ","
        << c.candidate_id_in_file << ","
        << c.is_pivot
        << "\n";
    }
    ofs.close();
    std::cout << "All candidates saved to " << filename << std::endl;

}


/**
 * Save a vector of Candidates to a CSV file, including how many are related, etc.
 */
void save_candidates_to_csv(const std::vector<Candidate>& pivots,
                            const std::string& filename) 
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }
    ofs << "snr,period,dm,acc,nh,ddm_count_ratio,ddm_snr_ratio,nassoc,"
        << "search_uuid,xml_file,candidate_id_in_file,num_related\n";




    // Write data rows
    for (const auto& c : pivots) {
        ofs << c.snr << ","
            << std::fixed << std::setprecision(17) << c.period << "," << std::fixed << std::setprecision(8)
            << c.dm << ","
            << c.acc << ","
            << c.nh << ","
            << c.ddm_count_ratio << ","
            << c.ddm_snr_ratio << ","
            << c.nassoc << ","
            << c.period_ms << ","
            << c.search_candidates_database_uuid << ","
            << c.xml_file_name << ","
            << c.candidate_id_in_file << ","
            << c.related_candidates.size()
            << "\n";
    }

    ofs.close();
    std::cout << "Pivot candidates saved to " << filename << std::endl;
}


/**
 * Analyse the clustered candidates. In this example, we simply “shortlist” them,
 * sort them by the size of their related_candidate list, and write them to CSV.
 */
void analyse_clusters(std::vector<Candidate>& candidates) {
    // Shortlist
    std::vector<Candidate> shortlisted_candidates = shortlist_candidates(candidates);

    std::cout << "Found " << shortlisted_candidates.size() << " pivot candidates.\n";

    // Sort pivot candidates by number of related candidates, descending
    std::sort(shortlisted_candidates.begin(),
              shortlisted_candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.related_candidates.size() > b.related_candidates.size();
              });

    // Optionally print some info or do other analysis
    // for (const auto& pivot : shortlisted_candidates) {
    //     std::cout << "Pivot Period=" << pivot.period << " DM=" << pivot.dm
    //               << " #Related=" << pivot.related_candidates.size() << std::endl;
    // }

    // Write pivot (shortlisted) candidates to CSV
    save_candidates_to_csv(shortlisted_candidates, "pivots.csv");
    save_all_candidates_to_csv(candidates, "all_candidates.csv");
}


int main(int argc, char** argv) {

    int opt;
    double period_thresh = 1e-7;
    double dm_thresh = 100;
    int ncpus = 180;
    std::vector<std::string> xml_files;

    // Parse options
    while ((opt = getopt(argc, argv, "p:d:n:")) != -1) {
        switch (opt) {
            case 'p':  // Period threshold
                period_thresh = std::atof(optarg);
                break;
            case 'd':  // DM threshold
                dm_thresh = std::atof(optarg);
                break;
            case 'n': // ncpus
                ncpus = std::atoi(optarg);
                break;
            case '?':  // Unknown option
                std::cerr << "Usage: " << argv[0] << " -p <period> -d <dm_thresh> -n <ncpus> file1.xml file2.xml ...\n";
                return 1;
        }
    }

    // Remaining arguments after option parsing are XML files
    for (int i = optind; i < argc; ++i) {
        xml_files.push_back(argv[i]);
    }

    // Validate input
    if (xml_files.empty()) {
        std::cerr << "Error: No XML files provided.\n";
        std::cerr << "Usage: " << argv[0] << " -p <period> -d <dm_thresh> file1.xml file2.xml ...\n";
        return 1;
    }

    // Print parsed values
    std::cout << "Period Threshold: " << period_thresh << "\n";
    std::cout << "DM Threshold: " << dm_thresh << "\n";
    std::cout << "XML files provided:\n";
    for (const auto& file : xml_files) {
        std::cout << " - " << file << '\n';
    }

    // // Example root dir; change as needed
    // std::string root_dir = "/b/PROCESSING/05_SEARCH/J0514-4002A/2024-05-19-15:50:23";

    // // 1) Get all XML files in the directory
    // std::vector<std::string> xml_files = get_xml_files(root_dir);

    // If you only want to parse the first N files (e.g., 10):
    // Make sure there are at least 10 files found, or guard accordingly.
    // if (xml_files.size() > 50) {
    //     xml_files.resize(50);
    // }

    omp_set_num_threads(ncpus);


    // 2) Parse all candidates (in parallel)
    std::vector<Candidate> all_candidates;

    #pragma omp parallel
    {
        std::vector<Candidate> local_candidates;
        #pragma omp for
        for (size_t i = 0; i < xml_files.size(); ++i) {
            std::cout << "Parsing " << xml_files[i] << std::endl;
            auto parsed = parse_candidates_from_file(xml_files[i]);
            local_candidates.insert(local_candidates.end(),
                                    parsed.begin(),
                                    parsed.end());
        }

        #pragma omp critical
        {
            all_candidates.insert(all_candidates.end(),
                                  local_candidates.begin(),
                                  local_candidates.end());
        }
    }

    std::cout << "Loaded total of " << all_candidates.size() << " candidates.\n";

    // 3) Sort by SNR descending, for example
    std::sort(all_candidates.begin(),
              all_candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.snr > b.snr;
              });

    // 4) Perform clustering
    cluster_candidates(all_candidates, period_thresh);

    // 5) Analyse the clusters, shortlist, etc. Also writes a CSV of pivots.
    analyse_clusters(all_candidates);

    return 0;
}
