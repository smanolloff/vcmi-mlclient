// =============================================================================
// Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "./agent-v5.h"
#include "AI/MMAI/common.h"
#include "AI/MMAI/schema/v5/types.h"
#include "AI/MMAI/schema/v5/constants.h"

namespace ML {
    namespace UserAgents {
        std::string AgentV5::getName() { return "UserAgent (v5)"; };
        int AgentV5::getVersion() { return 5; };
        double AgentV5::getValue(const MMAI::Schema::IState * s) { return -666; };

        MMAI::Schema::Action AgentV5::getAction(const MMAI::Schema::IState * s) {
            MMAI::Schema::Action act;

            if (s->version() != 5)
                throw std::runtime_error("Expected version 5, got: " + std::to_string(s->version()));

            auto any = s->getSupplementaryData();
            auto err = MMAI::Schema::AnyCastError(any, typeid(const MMAI::Schema::V5::ISupplementaryData*));
            ASSERT(err.empty(), "anycast for getSumpplementaryData error: " + err);

            auto sup = std::any_cast<const MMAI::Schema::V5::ISupplementaryData*>(any);
            auto side = static_cast<int>(sup->getSide());

            if (steps == 0 && benchmark) {
                t0 = clock();
            }

            steps++;

            if (sup->getType() == MMAI::Schema::V5::ISupplementaryData::Type::ANSI_RENDER) {
                std::cout << sup->getAnsiRender() << "\n";
                // use stored mask from pre-render result
                act = interactive
                    ? promptAction(lastmask, s)
                    : (actions.empty() ? randomValidAction(lastmask, s) : recordedAction());

                render = false;
            } else if (!benchmark && !render) {
                logAi->debug("Side: %d", side);
                render = true;
                // store mask of this result for the next action
                lastmask = s->getActionMask();
                act = MMAI::Schema::ACTION_RENDER_ANSI;
            } else if (sup->getIsBattleEnded()) {
                resets++;

                switch (resets % 4) {
                case 0: printf("\r|"); break;
                case 1: printf("\r\\"); break;
                case 2: printf("\r-"); break;
                case 3: printf("\r/"); break;
                }

                if (resets == 10) {
                    auto s = double(clock() - t0) / CLOCKS_PER_SEC;
                    printf("  steps/s: %-6.0f resets/s: %-6.2f\n", steps/s, resets/s);
                    resets = 0;
                    steps = 0;
                    t0 = clock();
                }

                std::cout.flush();

                if (!benchmark) logGlobal->debug("user-callback battle ended => sending ACTION_RESET");
                act = MMAI::Schema::ACTION_RESET;
            // } else if (false)
            } else {
                render = false;
                act = interactive
                    ? promptAction(s->getActionMask(), s)
                    : (actions.empty() ? randomValidAction(s->getActionMask(), s) : recordedAction());
            }

            if (verbose && !benchmark) logGlobal->debug("user-callback getAction returning: %d", EI(act));
            return act;
        };


        MMAI::Schema::Action AgentV5::promptAction(const MMAI::Schema::ActionMask &mask, const MMAI::Schema::IState * s) {
            int num;

            while (true) {
                // TODO
                std::cout << "NOT IMPLEMENTED: Enter 3 integers (primary action and hex); or 0 for a random valid action): ";

                // Read the user input as a string
                std::string input;
                std::getline(std::cin, input);

                // If the input is empty, treat it as if 0 was entered
                if (input.empty()) {
                    num = 0;
                    break;
                } else {
                    try {
                        num = std::stoi(input);
                        if (num >= 0)
                            break;
                        else
                            std::cerr << "Invalid input!\n";
                    } catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid input!\n";
                    } catch (const std::out_of_range& e) {
                        std::cerr << "Invalid input!\n";
                    }
                }
            }

            return num == 0 ? randomValidAction(mask, s) : MMAI::Schema::Action(num);
        }

        MMAI::Schema::Action AgentV5::recordedAction() {
            if (recording_i >= actions.size()) throw std::runtime_error("\n\n*** No more recorded actions in actions.txt ***\n\n");
            return MMAI::Schema::Action(actions.at(recording_i++));
        };

        MMAI::Schema::Action AgentV5::randomValidAction(const MMAI::Schema::ActionMask &mask, const MMAI::Schema::IState * s) {
            using PA = MMAI::Schema::V5::PrimaryAction;

            // // DEBUG - move to self (1-stack army only)
            // return (75 << 8) | EI(PA::MOVE);

            auto validPrimaryActions = std::vector<int>{};

            // start from (skip RETREAT)
            static_assert(EI(PA::RETREAT) == 0);
            for (int i = 1; i < EI(PA::_count); i++) {
                if (mask.at(i))
                    validPrimaryActions.push_back(i);
            }

            if (validPrimaryActions.empty()) {
                logAi->info("No valid primary actions => reset");
                return MMAI::Schema::ACTION_RESET;
            }

            int primaryAction;

            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dist(0, validPrimaryActions.size() - 1);
                int randomIndex = dist(gen);
                primaryAction = validPrimaryActions.at(randomIndex);
            }

            if (primaryAction < EI(PA::MOVE))
                // no hex needed
                return primaryAction;

            using MA = MMAI::Schema::V5::MiscAttribute;

            constexpr auto n0 = std::get<2>(MMAI::Schema::V5::MISC_ENCODING.at(EI(MA::PRIMARY_ACTION_MASK)));
            constexpr auto n1 = std::get<2>(MMAI::Schema::V5::MISC_ENCODING.at(EI(MA::SHOOTING)));
            static_assert(EI(MA::PRIMARY_ACTION_MASK) == 0);
            static_assert(EI(MA::SHOOTING) == 1);
            static_assert(n1 == 1);
            // either 0.0 or 1.0, but compare with 0.5
            // (to avoid floating-point issues)
            auto shooting = s->getBattlefieldState().at(n0) > 0.5;

            if (primaryAction > EI(PA::MOVE) && shooting)
                // no hex needed
                return primaryAction;

            using AMA = MMAI::Schema::V5::AMoveAction;
            auto validHexes = std::vector<int>{};

            for (int ihex = 0; ihex < 165; ihex++) {
                auto ibase = EI(PA::_count) + ihex * EI(AMA::_count);
                auto i = ibase + primaryAction - EI(PA::MOVE);
                if (mask.at(i))
                    validHexes.push_back(ihex);
            }

            int hex;

            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dist(0, validHexes.size() - 1);
                int randomIndex = dist(gen);
                hex = validHexes.at(randomIndex);
            }

            return (hex << 8) | primaryAction;
        }

        MMAI::Schema::Action AgentV5::firstValidAction(const MMAI::Schema::ActionMask &mask) {
            for (int j = 1; j < mask.size(); j++)
                if (mask.at(j)) return j;

            return -5;
        }
    }
}
