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
  int n, t;
  inputStream >> n >> t;
  string queue;
  inputStream >> queue;

  for (int j(0); j < t; j++) {
    for (int i(n - 1); i > 0;) {
      if (queue[i] == 'G' && queue[i - 1] == 'B') {
        swap(queue[i], queue[i - 1]);
        i -= 2;
      } else {
        i--;
      }
    }
  }
  cout << queue << endl;
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
