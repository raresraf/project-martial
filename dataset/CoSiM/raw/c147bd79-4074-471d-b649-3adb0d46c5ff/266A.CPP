#include <iostream>

using namespace std;

int main() {
  int n = 0, stones = 0;
  string color = "";
  cin >> n;
  cin >> color;
  for (int i = 0; i < color.length(); i++) {
    if (color[i] == color[i + 1]) {
      stones++;
    }
  }

  cout << stones;
  return 0;
}
