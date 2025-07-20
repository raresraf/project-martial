#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int testCases;
  string tooLong;

  cin >> testCases;

  while (testCases--) {
    cin >> tooLong;

    if (tooLong.size() > 10) {
      cout << *(tooLong.begin()) << (tooLong.length() - 2)
           << *(tooLong.end() - 1) << endl;
    } else {
      cout << tooLong << endl;
    }
  }

  return 0;
}
