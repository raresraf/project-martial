#include <iostream>
#include <string>

using namespace std;

int main() {
  int num;
  int x = 0;
  cin >> num;
  for (int i = 0; i < num; i++) {
    string X;
    cin >> X;
    if (X.compare("X++") == 0 || X.compare("++X") == 0) {
      x++;
    } else if (X.compare("X--") == 0 || X.compare("--X") == 0) {
      x--;
    }
  }
  cout << x;
  return 0;
}
