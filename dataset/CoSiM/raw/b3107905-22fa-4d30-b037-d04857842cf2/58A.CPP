#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {

  string text;
  string to_search = "hello";

  cin >> text;

  int pos_searching = 0;
  for (int i = 0; i < text.size(); i++) {
    // Character founded, next character.
    if (text[i] == to_search[pos_searching]) {
      pos_searching++;
    }
  }

  cout << (pos_searching == to_search.size() ? "YES" : "NO") << endl;
  return 0;
}
