
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

bool DEBUG = false;

ifstream fileStream;

istream &getInputStream() {
  if (DEBUG) {
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/'));
    path += "input.txt";
    fileStream.open(path);
    return fileStream;
  }
  return cin;
}

void solve() {
  istream &inputStream = getInputStream();

  // Your code goes here
  string expression;
  inputStream >> expression;
  char numbers[expression.length()];
  int numbersCount = 0;
  for (int i(0); i < expression.length(); i++) {
    if (expression[i] == '+') {
      continue;
    }
    numbers[numbersCount] = expression[i];
    numbersCount++;
  }
  sort(numbers, numbers + numbersCount);

  string result;
  for (int i(0); i < numbersCount; i++) {
    result += numbers[i];
    if (i != numbersCount - 1) {
      result += '+';
    }
  }
  cout << result << endl;
}

int main(int argc, char *argv[]) {
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  solve();
  return 0;
}
