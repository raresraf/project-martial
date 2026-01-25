#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int x = 0;
  while (n--) {
    string str;
    cin >> str;
    if (str[1] == '+')
      x++;
    else if (str[1] == '-')
      x--;
  }
  cout << x;
  return 0;
}