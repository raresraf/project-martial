#include <iostream>

using namespace std;

int main() {
  int n;
  int numberOfRemoves = 0;
  string row;

  cin >> n;
  cin.get();
  getline(cin, row);

  string::iterator previous = row.begin();
  string::iterator next = row.begin() + 1;

  while (next != row.end()) {
    if (*previous == *next) {
      next = row.erase(next);
      numberOfRemoves++;
    } else {
      previous++;
      next++;
    }
  }

  cout << numberOfRemoves;

  return 0;
}