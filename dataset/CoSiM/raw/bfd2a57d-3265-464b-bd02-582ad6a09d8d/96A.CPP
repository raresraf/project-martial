#include <iostream>
using namespace std;

void check(string x) {
  for (int i = 0; i < x.size(); i++) {
    if (x[i] == '0' && x[i + 1] == '0' && x[i + 2] == '0' && x[i + 3] == '0' &&
        x[i + 4] == '0' && x[i + 5] == '0' && x[i + 6] == '0') {
      cout << "YES";
      break;
    } else if (x[i] == '1' && x[i + 1] == '1' && x[i + 2] == '1' &&
               x[i + 3] == '1' && x[i + 4] == '1' && x[i + 5] == '1' &&
               x[i + 6] == '1') {
      cout << "YES";
      break;
    } else if (i == x.size() - 1) {
      cout << "NO";
      break;
    }
  }
}
int main() {
  string sen;
  cin >> sen;
  check(sen);
  void check();
  return 0;
}
