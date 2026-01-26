#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string input;
  getline(cin, input);

  transform(input.begin(), input.end(), input.begin(),
            ::tolower); // changing to lower case

  string output = "";

  for (int i = 0; i < input.length(); i++) {
    if (input.at(i) == 'a' || input.at(i) == 'o' || input.at(i) == 'y' ||
        input.at(i) == 'e' || input.at(i) == 'u' || input.at(i) == 'i') {
      continue;
    } else {
      output = output + "." + input.at(i);
    }
  }

  cout << output;

  return 0;
}
