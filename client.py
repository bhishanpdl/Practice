import random, requests, time

url = 'http://localhost:8000/{}'

def main():
    while True:
        print('woke up')
        requests.get(url.format('rule'))
        requests.get(url.format('command'))
        time.sleep(random.randint(2, 7))

if __name__ == '__main__':
    main()
