#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int resultado = 0;
  string colors;
  cin >> colors;
  for (int i = 0; i < colors.size(); i++) {
    if (colors[i] == colors[i + 1])
      resultado++;
  }
  cout << resultado;
  return 0;
}
