#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

int convert_to_digit(char a) { return a - 48; }

int main() {
  int x = 0;
  int n;
  cin >> n;
  int total = 0;
  while (n--) {
    char a, b, c;
    cin >> a >> b >> c;
    if (a == '+' && b == '+') {
      total += 1;
    } else if (a == '-' && b == '-') {
      total -= 1;
    } else if (b == '-' && c == '-') {
      total -= 1;
    } else if (b == '+' && c == '+') {
      total += 1;
    }
  }
  cout << total;
}