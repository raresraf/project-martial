#include <iostream>
#include <string>
using namespace std;
int main() {
  int first, second;
  int i;
  string firstline, secondline;
  cin >> firstline;
  cin >> secondline;
  for (i = 0; i < firstline.length(); i++) {
    first = (firstline[i] > 64 && firstline[i] < 91) ? firstline[i] - 64
                                                     : firstline[i] - 96;
    second = (secondline[i] > 64 && secondline[i] < 91) ? secondline[i] - 64
                                                        : secondline[i] - 96;
    if (first > second) {
      cout << 1 << endl;
      break;
    } else if (first < second) {
      cout << -1 << endl;
      break;
    }
  }
  if (i == firstline.length())
    cout << 0 << endl;
  return 0;
}
