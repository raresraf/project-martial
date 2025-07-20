#include <iostream>
#include <limits>
#include <string>

using namespace std;

int main() {
  int x{}, n{};
  cin >> n;
  cin.ignore(numeric_limits<streamsize>::max(), '\n'); /* cleaning the buffer */

  for (int i = 0; i < n; i++) {
    string linha{};
    getline(cin, linha);
    if (linha[1] == '+')
      x++;
    if (linha[1] == '-')
      x--;
  }

  cout << x;
  return 0;
}