
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

  string first, second;
  inputStream >> first >> second;
  int32_t result = 0;
  for (int i(0); i < first.length(); i++) {
    char a = tolower(first[i]);
    char b = tolower(second[i]);
    if (a > b) {
      result = 1;
      break;
    } else if (a < b) {
      result = -1;
      break;
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
