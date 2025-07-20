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

  string word;
  int wordLength;
  for (int i(0); i < n; i++) {
    inputStream >> word;
    wordLength = word.length();
    if (wordLength <= 10) {
      cout << word << endl;
      continue;
    }

    word = word[0] + to_string(wordLength - 2) + word[wordLength - 1];
    cout << word << endl;
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