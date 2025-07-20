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
  int k, n, w;
  inputStream >> k >> n >> w;

  int dollarsNeeded = 0;
  for (int i(1); i <= w; i++) {
    dollarsNeeded += k * i;
  }

  cout << max(0, dollarsNeeded - n) << endl;
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
