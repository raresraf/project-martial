#include <iostream>
using namespace std;

int main() {

  int n, t;
  string str;
  cin >> n >> t;
  cin >> str;

  for (int j = 0; j < t; j++) {
    for (int i = 0; i < n;) {
      if (str[i] == 'B' && str[i + 1] == 'G') {
        str[i] = 'G';
        str[i + 1] = 'B';
        i = i + 2;
      } else
        i++;
    }
  }

  cout << str << endl;

  return 0;
}
