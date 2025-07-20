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

  int n;
  inputStream >> n;
  int numProblems = 0;
  for (int i(0); i < n; i++) {
    int count = 0;
    for (int j(0); j < 3; j++) {
      int solved;
      inputStream >> solved;
      count += solved;
    }
    if (count >= 2) {
      numProblems++;
    }
  }
  cout << numProblems;
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