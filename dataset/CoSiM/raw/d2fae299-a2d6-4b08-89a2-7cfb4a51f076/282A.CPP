#include <iostream>
#include <string>
using namespace std;
int main() {
  string str;
  int n, x = 0;
  cin >> n;
  for (int i = 0; i <= n; i++) {
    getline(cin, str);
    if (str[0] == 'X' && str[1] == '+' && str[2] == '+')
      x++;
    else if (str[0] == '+' && str[1] == '+' && str[2] == 'X')
      ++x;
    else if (str[0] == 'X' && str[1] == '-' && str[2] == '-')
      x--;
    else if (str[0] == '-' && str[1] == '-' && str[2] == 'X')
      --x;
    str.erase();
  }
  cout << x;
}
