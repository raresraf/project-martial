#include <iostream>
#include <string>

using namespace std;

int main() {

  int n;
  string word;
  int to_take = 0;
  cin >> n >> word;
  for (int i = 1; i < n; i++) {
    if (word[i] == word[i - 1]) {
      to_take++;
    }
  }

  cout << to_take << endl;

  return 0;
}
