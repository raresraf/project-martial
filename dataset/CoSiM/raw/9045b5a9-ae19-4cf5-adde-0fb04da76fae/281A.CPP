#include <iostream>
using namespace std;
int main() {
  string str1;
  cin >> str1;
  if (str1[0] >= 97 && str1[0] <= 122) {
    str1[0] = str1[0] - 32;
  }
  cout << str1;
  return 0;
}