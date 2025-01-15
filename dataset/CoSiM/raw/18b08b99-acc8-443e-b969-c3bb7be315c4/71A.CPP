#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  while (n--) {
    string word;
    cin >> word;
    if (word.size() <= 10) {
      cout << word << endl;
      continue;
    }

    cout << word.front() << word.size() - 2 << word.back();
    if (n > 0)
      cout << endl;
  }
  return 0;
}
