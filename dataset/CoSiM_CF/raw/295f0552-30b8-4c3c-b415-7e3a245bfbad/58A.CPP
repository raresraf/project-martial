#include <iostream>
#include <string>
using namespace std;

int main() {
  char hello[]{'h', 'e', 'l', 'l', 'o'};
  string str;
  int j{0}, sum{0};
  cin >> str;
  for (size_t i{0}; i < str.length(); i++) {
    if (str[i] == hello[j]) {
      j++;
      sum++;
    }
    if (sum == 5)
      break;
  }
  if (sum == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
