#include "llm.h"



static const char *usage_message = R"(
USAGE:
./smollm2 [options] -p PROMPT  for a single prompt or
./smollm2 [options] for a chat interface. 

Optional args.
--llm MODEL_SIZE:  The LLM to use to respond to prompt. MODEL_SIZE options are (small, medium, large)[default=small].
--npred  N : Max context size. Minimum is 128 and max is 8192 [default=512]. Higher values consume more memory.
)";


int main(int argc, char const *argv[])
{
    using namespace llm;

    const char* model_name = "smollm2-sm.bin";
    SmolLM2Type model_type = SmolLM2Type::Small;
    int max_ctx = 512;
    std::string prompt = "";

    for (int i = 1; i < argc; i++) {
        std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            fprintf(stderr, "%s\n.", usage_message);
            return 0;
        }
        else if (arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[i + 1];
                i += 1; // fast-forward
            } else {
                fprintf(stderr, "error: Prompt not provided.\n");
                fprintf(stderr, "%s\n.", usage_message);
                return -1;
            }
        }
        else if (arg == "--llm") {
            if (i + 1 < argc) {
                const std::string_view llm_arg{argv[i + 1]};
                if (llm_arg == "small") {
                    model_name = "smollm2-sm.bin";
                    model_type = llm::SmolLM2Type::Small;
                }
                else if (llm_arg == "medium") {
                    model_name = "smollm2-md.bin";
                    model_type = llm::SmolLM2Type::Medium;
                }
                else if (llm_arg == "large") {
                    model_name = "smollm2-lg.bin";
                    model_type = llm::SmolLM2Type::Large;
                } else {
                    printf("error: invalid llm option: %s.\n", llm_arg.data());
                    printf("%s\n", usage_message);
                    return -1;
                }
                i += 1; // fast-forward
            } else {
                printf("error: llm option is not provided.\n");
                printf("%s\n", usage_message);
                return -1;
            }
        }
        else if (arg == "--npred") {
            if (argc <= i+1) {
                fprintf(stderr, "npred value is missing.\n");
                return -1;
            }
            int npred;
            try {
                npred = std::stoi(argv[i+1]);
            } catch (...) {
                fprintf(stderr, "Invalid npred value.\n");
                return -1;
            }
            if (npred < 128 || npred > 8192) {
                fprintf(stderr, "npred must be greater than 128 and less than 2048.\n");
                return -1;
            }
            max_ctx = npred;
            i += 1; // skip len param
        }
        else {
            fprintf(stderr, "error: Unknown argument: %s\n", arg.data());
            fprintf(stderr, "%s\n.", usage_message);
            return -1;
        }
    }

    const std::string download_command = get_model_download_command(model_name);
    const int res = std::system(download_command.c_str());
    if (res != 0) {
        fprintf(stderr, "Error: Failed to download the models. Check your network connectivity.\n");
        return -1;
    }

    SmolLM2 model;
    smollm2_init(model, model_type, max_ctx, get_model_path(model_name));

    const int top_k = 40;
    const float top_p = 0.95;
    const float temp = 0.8;

    if (prompt == "") {
        printf("Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n");
        std::string prompt;
        while (true) {
            printf("\n\n[You]: "); fflush(stdout);

            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            printf("\n\n[SmolLM2]: \n"); fflush(stdout);
            
            topk_sample(model, prompt, top_k, top_p, temp);
        } 
    } else {
        printf("\n[PROMPT]:\n%s\n\n[SmolLM2]: ", prompt.c_str());
        std::fflush(stdout);

        const int processed_toks = topk_sample(model, prompt, top_k, top_p, temp);
    }

    smollm2_uninit(model);

    return 0;
}

