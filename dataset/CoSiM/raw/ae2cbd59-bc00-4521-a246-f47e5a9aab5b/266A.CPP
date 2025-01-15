#include <iostream>

using namespace std;

int main() {

  string s = "";
  int sum = 0;
  int n;

  cin >> n;
  cin.ignore();
  getline(cin, s);

  if (s.length() == 1) {
    sum = 0;
  }

  else {

    for (int i = 1; i <= s.length() - 1; ++i) {

      if (s[i - 1] == s[i]) {

        sum += 1;
      }
    }
  }

  cout << sum;

  return 0;
}
