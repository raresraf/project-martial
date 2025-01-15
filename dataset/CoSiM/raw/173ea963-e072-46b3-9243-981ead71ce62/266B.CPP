#include <iostream>

using namespace std;

int main() {

  string str;
  int n;
  int t;
  char tmp;

  cin >> n >> t;
  cin.ignore();
  getline(cin, str);

  for (int i = 1; i <= t; ++i) {

    for (int j = 0; j < n - 1; ++j) {
      if ((int)str[j] < (int)str[j + 1]) {
        tmp = str[j];
        str[j] = str[j + 1];
        str[j + 1] = tmp;
        j += 1;
      }
    }
  }

  cout << endl << str;

  return 0;
}
