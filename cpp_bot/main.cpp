#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <tgbot/tgbot.h>
#include <map>
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

std::map<long long int, bool> waitingForPhoto;

size_t write_callback(char *contents, size_t size, size_t nmemb, std::string *userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    const char* token_env = getenv("TELEGRAM_BOT_TOKEN");
    if (token_env == nullptr) {
        std::cerr << "Ошибка: переменная окружения TELEGRAM_BOT_TOKEN не установлена." << std::endl;
        return 1;
    }
    std::string token(token_env);

    const char* server_url_env = getenv("SERVER_URL");
    if (server_url_env == nullptr) {
        std::cerr << "Ошибка: переменная окружения SERVER_URL не установлена." << std::endl;
        return 1;
    }
    const std::string serverUrl(server_url_env);
    TgBot::Bot bot(token);

    std::vector<TgBot::BotCommand::Ptr> commands;
    TgBot::BotCommand::Ptr commandStart(new TgBot::BotCommand);
    commandStart->command = "start";
    commandStart->description = "Начать работу с ботом";
    commands.push_back(commandStart);
    bot.getApi().setMyCommands(commands);

    TgBot::InlineKeyboardMarkup::Ptr startKeyboard(new TgBot::InlineKeyboardMarkup);
    std::vector<TgBot::InlineKeyboardButton::Ptr> startKeyboardRow1;
    TgBot::InlineKeyboardButton::Ptr identifyButton(new TgBot::InlineKeyboardButton);
    identifyButton->text = "✅ Определить болезнь по фото";
    identifyButton->callbackData = "identify_illness";
    startKeyboardRow1.push_back(identifyButton);
    startKeyboard->inlineKeyboard.push_back(startKeyboardRow1);

    std::vector<TgBot::InlineKeyboardButton::Ptr> startKeyboardRow2;
    TgBot::InlineKeyboardButton::Ptr aboutDevelopersButton(new TgBot::InlineKeyboardButton);
    aboutDevelopersButton->text = "ℹ️ О разработчиках";
    aboutDevelopersButton->callbackData = "about";
    startKeyboardRow2.push_back(aboutDevelopersButton);
    startKeyboard->inlineKeyboard.push_back(startKeyboardRow2);

    TgBot::InlineKeyboardMarkup::Ptr afterPhotoKeyboard(new TgBot::InlineKeyboardMarkup);
    std::vector<TgBot::InlineKeyboardButton::Ptr> afterPhotoKeyboardRow1;
    afterPhotoKeyboardRow1.push_back(identifyButton);
    afterPhotoKeyboard->inlineKeyboard.push_back(afterPhotoKeyboardRow1);

    bot.getEvents().onCommand("start", [&bot, &startKeyboard](TgBot::Message::Ptr message) {
        std::string text = "Привет! 🌿\n\nЯ ваш личный помощник-садовод.\n\nОтправьте мне фотографию вашего растения, и я определю его болезнь и дам рекомендации по лечению.\n\nНажмите на кнопку ниже, чтобы начать.";
        bot.getApi().sendMessage(message->chat->id, text, nullptr, 0, startKeyboard);
    });

    bot.getEvents().onCallbackQuery([&bot, &startKeyboard](TgBot::CallbackQuery::Ptr query) {
        if (query->data == "identify_illness") {
            bot.getApi().sendMessage(query->message->chat->id, "Теперь, пожалуйста, отправьте мне фотографию пораженного растения. Постарайтесь сделать снимок четким и при хорошем освещении.");
            waitingForPhoto[query->message->chat->id] = true;
        } else if (query->data == "about") {
            bot.getApi().sendMessage(query->message->chat->id, "Бот разработан командой \"Кто. Мы\"");
        }
    });

    bot.getEvents().onAnyMessage([&bot, &afterPhotoKeyboard, &serverUrl](TgBot::Message::Ptr message) {
        if (message->text.length() > 0 && StringTools::startsWith(message->text, "/start")) {
            return;
        }
        long long int chatId = message->chat->id;
        if (waitingForPhoto.count(chatId) && waitingForPhoto[chatId]) {
            if (!message->photo.empty()) {
                waitingForPhoto[chatId] = false;
                
                bot.getApi().sendMessage(chatId, "Фото получено, начинаю анализ... ⏳");

                TgBot::PhotoSize::Ptr photo = message->photo.back();
                TgBot::File::Ptr file = bot.getApi().getFile(photo->fileId);
                std::string fileContent = bot.getApi().downloadFile(file->filePath);

                CURL *curl;
                CURLcode res;
                std::string readBuffer;
                long http_code = 0;

                curl_global_init(CURL_GLOBAL_DEFAULT);
                curl = curl_easy_init();
                if (curl) {
                    curl_mime *form = curl_mime_init(curl);
                    curl_mimepart *field = curl_mime_addpart(form);

                    curl_mime_name(field, "image");
                    curl_mime_filename(field, "photo.jpg");
                    curl_mime_data(field, fileContent.data(), fileContent.size());

                    curl_easy_setopt(curl, CURLOPT_URL, serverUrl.c_str());
                    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

                    res = curl_easy_perform(curl);
                    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
                    
                    std::cout << "HTTP код ответа: " << http_code << std::endl;

                    if (res == CURLE_OK && http_code == 200) {
                        std::cout << "Сервер ответил: " << readBuffer << std::endl;
                        try {
                            auto jsonResponse = nlohmann::json::parse(readBuffer);
                            std::string diseaseName = jsonResponse.value("disease_name", "Неизвестно");
                            std::stringstream response_ss;
                            if (diseaseName == "Болезнь не распознана") {
                                response_ss << "🔍 *Результат анализа: Болезнь не распознана* 😞\n\n";
                                response_ss << "К сожалению, мне не удалось определить болезнь по этой фотографии. "
                                            << "Пожалуйста, попробуйте сделать снимок более четким, при хорошем освещении, "
                                            << "сфокусировавшись на пораженной части растения. Иногда это помогает!\n\n";
                                response_ss << "❗️ *Важная памятка:*\n";
                                response_ss << "_" << jsonResponse.value("memo", "Информация отсутствует.") << "_\n";
                            } else {
                                response_ss << "🔍 *Результат анализа: " << diseaseName << "*\n\n";

                                if (jsonResponse.contains("treatment") && !jsonResponse["treatment"].empty()) {
                                    response_ss << "💊 *Рекомендации по лечению:*\n";
                                    for (const auto& step : jsonResponse["treatment"]) {
                                        response_ss << "• " << step.get<std::string>() << "\n";
                                    }
                                    response_ss << "\n";
                                }
                                if (jsonResponse.contains("prevention") && !jsonResponse["prevention"].empty()) {
                                    response_ss << "🛡️ *Профилактика:*\n";
                                    for (const auto& step : jsonResponse["prevention"]) {
                                        response_ss << "• " << step.get<std::string>() << "\n";
                                    }
                                    response_ss << "\n";
                                }
                                if (jsonResponse.contains("memo") && !jsonResponse["memo"].empty()) {
                                    response_ss << "❗️ *Важная памятка:*\n";
                                    response_ss << "_" << jsonResponse["memo"].get<std::string>() << "_\n";
                                }
                            }
                            bot.getApi().sendMessage(chatId, response_ss.str(), nullptr, 0, afterPhotoKeyboard, "Markdown");
                        } catch (const nlohmann::json::parse_error& e) {
                            std::cerr << "Ошибка парсинга JSON: " << e.what() << std::endl;
                            bot.getApi().sendMessage(chatId, "Упс! Произошла внутренняя ошибка. Не удалось обработать ответ от сервера. Попробуйте позже.");
                        }
                    } else {
                        std::cerr << "Сервер вернул ошибку (код " << http_code << "): " << readBuffer << std::endl;
                        bot.getApi().sendMessage(chatId, "Упс! Сервер анализа изображений сейчас недоступен. Пожалуйста, попробуйте позже.");
                    }

                    curl_mime_free(form);
                    curl_easy_cleanup(curl);
                }
                curl_global_cleanup();
            } else {
                bot.getApi().sendMessage(chatId, "Кажется, это не фотография. Пожалуйста, отправьте именно фото.");
            }
        } else {
            if(message->text.length() > 0) {
                bot.getApi().sendMessage(chatId, "Я не знаю, как на это ответить. Чтобы начать, используйте команду /start или нажмите на кнопку \"Определить болезнь по фото\".");
            }
        }
    });
    signal(SIGINT, [](int s) {
        std::cout << "SIGINT получено, бот останавливается.\n";
        exit(0);
    });

    try {
        printf("Бот запущен. Имя пользователя: %s\n", bot.getApi().getMe()->username.c_str());
        bot.getApi().deleteWebhook();

        TgBot::TgLongPoll longPoll(bot);
        while (true) {
            longPoll.start();
        }
    } catch (std::exception& e) {
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
    }
    return 0;
}