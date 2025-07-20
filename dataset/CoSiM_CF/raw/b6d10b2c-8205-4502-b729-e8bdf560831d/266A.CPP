#include <iostream>
using namespace std;

int main() {
  string str;
  int n, num = 0, i;

  cin >> n;
  cin >> str;
  for (i = 0; i < n - 1; i++) {
    if (str[i] == str[i + 1]) {
      num += 1;
    }
  }
  cout << num;

  return 0;
}