#include <iostream>
#include <string>
#include <unordered_set>

using namespace std;

int main() {
  string input, output = "";
  cin >> input;
  unordered_set<char> vowels = {'a', 'i', 'u', 'e', 'o', 'y'};
  for (int i = 0; i != input.size(); i++) {
    if (vowels.find(tolower(input[i])) == vowels.end()) {
      output += '.';
      output += tolower(input[i]);
    }
  }

  cout << output;
}
