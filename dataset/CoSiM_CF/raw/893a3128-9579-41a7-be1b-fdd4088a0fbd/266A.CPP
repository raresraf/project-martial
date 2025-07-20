#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, minim = 0;
  string color;
  cin >> n >> color;
  for (int i = 0; i < (n - 1); i++) {
    if (color.at(i) == color.at(i + 1))
      minim++;
  }

  cout << minim;
}
