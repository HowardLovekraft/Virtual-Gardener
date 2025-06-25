# Деплой модели
## Предварительные требования
Перед началом работы убедитесь, что на вашей системе установлены:
- Docker
- Docker Compose

## Установка и запуск
Следуйте этим шагам для развертывания проекта.

1. Клонируйте репозиторий

Откройте терминал и выполните следующие команды:  
`git clone https://github.com/HowardLovekraft/Virtual-Gardener.git`  
`cd Virtual-Gardener`

Также скачайте [веса модели](https://drive.google.com/file/d/1qYk3Ls1HW0WHZSAbUl-c7XGO5uRKkgPj/view?usp=sharing) и скопируйте их по пути  
 `Virtual-Gardener/model/YOLO_files/trained_models/v0.1/`

2. Создайте файл конфигурации

Для работы бота необходим файл `.env` в корневой папке проекта. Он содержит токен вашего Telegram-бота.

Создайте файл с именем .env и добавьте в него:
```
DATASET_DIR=/app/New-Plant-Diseases-Dataset(Augmented)
PRETRAINED_MODEL=/app/model/YOLO_files/trained_models/v0.1/best.pt
MODEL_YAML=/app/model/YOLO_files/yolov8-cls.yaml
TELEGRAM_BOT_TOKEN=ВАШ_ТЕЛЕГРАМ_ТОКЕН_ЗДЕСЬ
```

Замените `ВАШ_ТЕЛЕГРАМ_ТОКЕН_ЗДЕСЬ` на реальный токен, который вы получили от @BotFather.

3. Соберите и запустите проект

Выполните одну команду в терминале, находясь в корневой папке проекта:  
`docker-compose up --build`  
`docker-compose up`: эта команда запускает все сервисы (py\_server и cpp\_bot), описанные в файле docker-compose.yml.

## Как пользоваться ботом

Найдите вашего бота в Telegram по его имени пользователя.  
Отправьте команду `/start`.  
Нажмите на кнопку "✅ Определить болезнь по фото".  
Отправьте фотографию листа растения.  
Дождитесь ответа с результатом анализа и рекомендациями.

## Остановка проекта

Чтобы остановить работу всех сервисов и удалить запущенные контейнеры, нажмите `Ctrl + C` в терминале, где запущен docker-compose, или откройте новый терминал в той же папке и выполните команду:  
`docker-compose down`
