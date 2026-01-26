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

  int n;
  inputStream >> n;

  int exitNum, enterNum;
  int passengers = 0, capacity = 0;
  for (int i(0); i < n; i++) {
    inputStream >> exitNum >> enterNum;
    passengers = passengers - exitNum + enterNum;
    if (passengers > capacity) {
      capacity = passengers;
    }
  }
  cout << capacity << endl;
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
