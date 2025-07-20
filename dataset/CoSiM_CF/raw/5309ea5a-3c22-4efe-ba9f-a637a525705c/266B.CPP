#include <iostream>
using namespace std;
void linemover(string &m) {
  int len = m.size();
  int index = 0;
  while (index <= len - 1) {
    if (m[index] == 'B' && m[index + 1] == 'G') {
      // swap
      m[index] = 'G';
      m[index + 1] = 'B';
      index = index + 2;
    } else {
      index = index + 1;
    }
  }
}

int main() {
  string line;
  int total;
  int time;
  cin >> total;
  cin >> time;
  cin >> line;
  for (int i = 0; i < time; i++) {
    linemover(line);
  }
  cout << line << endl;
}
