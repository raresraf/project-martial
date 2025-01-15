#include <iostream>
#include <string>

using namespace std;

int main() {
  int x = 0, n;
  string statement;
  cin >> n;
  while (n--) {
    cin >> statement;
    if (statement == "X++" || statement == "++X") {
      x++;
    }
    if (statement == "X--" || statement == "--X") {
      x--;
    }
  }
  printf("%d\n", x);
  return 0;
}
