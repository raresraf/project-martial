#include <iostream>
using namespace std;

int main() {
  int n, total = 0;
  char t;
  string kata;
  cin >> n;
  cin >> kata;

  for (int i = 0; i < n; i++) {
    if (i > 0) {
      if (kata[i] == t) {
        total++;
      }
    }
    t = kata[i];
  }
  cout << total;
  return 0;
}
