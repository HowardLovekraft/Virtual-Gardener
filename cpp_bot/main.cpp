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
    std::string token(token_env);
    const char* server_url_env = getenv("SERVER_URL");
    const std::string serverUrl(server_url_env);

    TgBot::Bot bot(token);

    std::vector<TgBot::BotCommand::Ptr> commands;
    TgBot::BotCommand::Ptr commandStart(new TgBot::BotCommand);
    commandStart->command = "start";
    commandStart->description = "запуск бота";
    commands.push_back(commandStart);
    bot.getApi().setMyCommands(commands);

    TgBot::InlineKeyboardMarkup::Ptr startKeyboard(new TgBot::InlineKeyboardMarkup);
    std::vector<TgBot::InlineKeyboardButton::Ptr> startKeyboardstr1;
    TgBot::InlineKeyboardButton::Ptr send_photoButton(new TgBot::InlineKeyboardButton);
    send_photoButton->text = "Отправить фото";
    send_photoButton->callbackData = "send";
    startKeyboardstr1.push_back(send_photoButton);
    TgBot::InlineKeyboardButton::Ptr about_developersButton(new TgBot::InlineKeyboardButton);
    about_developersButton->text = "О разработчиках";
    about_developersButton->callbackData = "about";
    startKeyboardstr1.push_back(about_developersButton);
    startKeyboard->inlineKeyboard.push_back(startKeyboardstr1);

    TgBot::InlineKeyboardMarkup::Ptr afterPhotoKeyboard(new TgBot::InlineKeyboardMarkup);
    std::vector<TgBot::InlineKeyboardButton::Ptr> afterPhotoKeyboardstr1;
    TgBot::InlineKeyboardButton::Ptr finish(new TgBot::InlineKeyboardButton);
    finish->text = "Завершить";
    finish->callbackData = "finish";
    afterPhotoKeyboardstr1.push_back(finish);
    afterPhotoKeyboardstr1.push_back(send_photoButton);
    afterPhotoKeyboard->inlineKeyboard.push_back(afterPhotoKeyboardstr1);

    bot.getEvents().onCommand("start", [&bot, &startKeyboard](TgBot::Message::Ptr message) {
        bot.getApi().sendMessage(message->chat->id,
                                    "Привет, я бот, определяющий болезнь домашнего растения по фото",
                                    nullptr,
                                    0,
                                    startKeyboard);
    });

    bot.getEvents().onCallbackQuery([&bot, &startKeyboard, &afterPhotoKeyboard](TgBot::CallbackQuery::Ptr query){
        if (query->data == "send"){
            bot.getApi().sendMessage(query->message->chat->id, "Пожалуйста, отправьте фото.");
            waitingForPhoto[query->message->chat->id] = true;
        }
        else if (query->data == "about") {
            bot.getApi().sendMessage(query->message->chat->id, "Бот разработан командой \"Кто. Мы\"");
        }
        else if (query->data == "finish") {
            waitingForPhoto.clear();
            return;
        }
    });

    bot.getEvents().onAnyMessage([&bot, &afterPhotoKeyboard, &serverUrl](TgBot::Message::Ptr message){
        if (StringTools::startsWith(message->text, "/start")) {
            waitingForPhoto.clear();
            return;
        }
        long long int chatId = message->chat->id;
        if (waitingForPhoto.count(chatId)){
            if (!message->photo.empty()){
                TgBot::PhotoSize::Ptr photo = message->photo.back();
                TgBot::File::Ptr file = bot.getApi().getFile(photo->fileId);
                std::string filePath = file->filePath;
                std::string fileExtension = filePath.substr(filePath.find_last_of('.'));
                std::string fileContent = bot.getApi().downloadFile(filePath);

                CURL *curl;
                CURLcode res;
                std::string readBuffer;
                long http_code = 0;

                curl_global_init(CURL_GLOBAL_DEFAULT);
                curl = curl_easy_init();
                if(curl) {
                    curl_easy_setopt(curl, CURLOPT_URL, serverUrl.c_str());
                    curl_easy_setopt(curl, CURLOPT_POST, 1L);

                    curl_mime *form = NULL;
                    curl_mimepart *field = NULL;
                    form = curl_mime_init(curl);

                    std::string fileName = "photo" + photo->fileId + fileExtension;

                    field = curl_mime_addpart(form);
                    curl_mime_name(field, "image");
                    curl_mime_filename(field, fileName.c_str());
                    curl_mime_type(field, "image/jpeg");
                    curl_mime_data(field, fileContent.data(), fileContent.size());
                    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
                    res = curl_easy_perform(curl);
                    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
                    std::cout << "HTTP код ответа: " << http_code << std::endl;

                    if(res == CURLE_OK) {
                        std::cout << "Сервер ответил: " << readBuffer << std::endl;
                        if (http_code == 200) {
                            try {
                                auto jsonResponse = nlohmann::json::parse(readBuffer);
                                std::string className = jsonResponse["class_name"].get<std::string>();
                                bot.getApi().sendMessage(message->chat->id, "Предсказанный класс: " + className, nullptr, 0, afterPhotoKeyboard);
                            } catch (const nlohmann::json::parse_error& e) {
                                std::cerr << "Ошибка парсинга JSON: " << e.what() << std::endl;
                            }
                        } else {
                            std::cerr << "Сервер вернул ошибку (код " << http_code << "): " << readBuffer << std::endl;
                        }
                    } else {
                        std::cerr << "Ошибка отправки запроса на сервер: " << curl_easy_strerror(res) << std::endl;
                    }
                    curl_mime_free(form);
                    curl_easy_cleanup(curl);
                }
                curl_global_cleanup();
            } else{
                bot.getApi().sendMessage(message->chat->id, "не тот тип файла, отправьте фото");
            }
        } else {
            bot.getApi().sendMessage(message->chat->id, "это не сработает, нажимай на предложенные кнопки или запусти бота заново");
        }
    });
    signal(SIGINT, [](int s) {
        std::cout << "SIGINT got\n";
        exit(0);
    });

    try {
        printf("Bot username: %s\n", bot.getApi().getMe()->username.c_str());
        bot.getApi().deleteWebhook();

        TgBot::TgLongPoll longPoll(bot);
        while (true) {
            std::cout << "Long poll started\n";
            longPoll.start();
        }
    } catch (std::exception& e) {
        std::cout << "error:\n" << e.what();
    }
}