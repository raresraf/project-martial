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

  int n, k;
  inputStream >> n >> k;
  int numAdvance = 0;
  int tmp;
  int numK = 0;
  for (int i(0); i < n; i++) {
    inputStream >> tmp;
    if (tmp <= 0) {
      break;
    }
    if (i < k - 1) {
      numAdvance++;
    } else if (i == k - 1) {
      numK = tmp;
      numAdvance++;
    } else if (tmp >= numK) {
      numAdvance++;
    } else {
      break;
    }
  }

  cout << numAdvance;
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