#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  int x = 0;
  while (n--) {
    string str;
    cin >> str;
    if (str[1] == '+')
      x++;
    else
      x--;
  }
  cout << x;

  return 0;
}