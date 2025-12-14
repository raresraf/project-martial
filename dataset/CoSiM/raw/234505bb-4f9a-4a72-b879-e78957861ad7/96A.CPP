#include <iostream>
using namespace std;

int main() {
  int n, total = 1;
  char t;
  string kata;
  cin >> kata;

  for (int i = 0; i < kata.length(); i++) {
    if (i > 0) {
      if (kata[i] == t) {
        total++;
      } else {
        total = 1;
      }
      if (total == 7) {
        break;
      }
    }
    t = kata[i];
  }
  if (total == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
