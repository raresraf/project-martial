#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>

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
  string name;
  inputStream >> name;
  set<char> uniqueChars;
  for (int i(0); i < name.length(); i++) {
    uniqueChars.insert(name[i]);
  }

  cout << (((uniqueChars.size() & 1) == 1) ? "IGNORE HIM!" : "CHAT WITH HER!");
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
