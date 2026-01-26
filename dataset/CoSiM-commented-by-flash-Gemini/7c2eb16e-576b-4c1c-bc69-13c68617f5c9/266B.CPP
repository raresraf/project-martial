#include <iostream>
using namespace std;

int main() {
  int am, ch;
  cin >> am >> ch;
  char st[am];
  cin >> st;
  for (; ch > 0; ch--) {
    for (int i = 0; i < am; i++) {
      if (st[i] == 'G' && st[i - 1] == 'B' && i != 0) {
        st[i] = 'B';
        st[i - 1] = 'G';
        if (st[i + 1] == 'G')
          i = i + 1;
      }
    }
  }
  cout << st;
  return 0;
}