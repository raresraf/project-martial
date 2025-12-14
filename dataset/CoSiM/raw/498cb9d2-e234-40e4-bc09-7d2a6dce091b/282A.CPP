#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  int x = 0;
  string command;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> command;
    if (command.compare("++X") == 0 || command.compare("X++") == 0) {
      x++;
    } else if (command.compare("--X") == 0 || command.compare("X--") == 0) {
      x--;
    }
  }
  cout << x << endl;
}

// g++ -std=c++11 -O2 -Wall 282A.cpp -o out