
#include <cmath>
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
  int element;
  for (int i(0); i < 5; i++) {
    for (int j(0); j < 5; j++) {
      inputStream >> element;
      if (element == 0) {
        continue;
      }
      cout << abs(2 - i) + abs(2 - j) << endl;
      return;
    }
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
