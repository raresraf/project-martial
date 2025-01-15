#include <iostream>

using namespace std;
int main() {
  string situation;
  cin >> situation;

  int consecutive = 0;
  int counter = 1;
  for (int i = 0; i < situation.size() - 1; i++) {
    if (situation.at(i) == situation.at(i + 1)) {
      counter++;
      if (counter >= consecutive)
        consecutive = counter;
    } else
      counter = 1;
  }

  if (consecutive >= 7)
    cout << "YES";
  else
    cout << "NO";
}
