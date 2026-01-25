#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, x = 0;
  cin >> n;
  string sta;
  while (n--) {
    cin >> sta;
    if (sta[1] == '+')
      x++;
    else
      x--;
  }
  cout << x << endl;
  return 0;
}
