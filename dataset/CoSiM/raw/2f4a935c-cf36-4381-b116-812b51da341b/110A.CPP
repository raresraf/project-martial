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
  unsigned long long number;
  inputStream >> number;

  int luckyNumbers = 0;
  while (number > 0) {
    if (number % 10 == 4 || number % 10 == 7) {
      luckyNumbers++;
    }
    number /= 10;
  }
  if (luckyNumbers == 7 || luckyNumbers == 4) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
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
