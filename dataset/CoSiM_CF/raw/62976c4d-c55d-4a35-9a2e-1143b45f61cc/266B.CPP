#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int n, t;
  string str;
  char temp;
  cin >> n >> t;
  cin >> str;

  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (str[j] == 'B' && str[j + 1] == 'G') {
        temp = str[j + 1];
        str[j + 1] = str[j];
        str[j] = temp;
        j++;
      }
    }
  }

  cout << str << endl;

  return 0;
}
