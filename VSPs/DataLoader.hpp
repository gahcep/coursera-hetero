#pragma once

#include <limits>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <type_traits>

namespace DataLoader
{
	enum class Kind 
	{
		Arbitrary,
		Vector,
		Matrix
	};

	namespace Flags
	{
		// No Flags
		const unsigned NullOpts = 0;

		namespace Arbitrary
		{
			/*
			-- Work with arbitrary values/numbers
			*/

			namespace Parse
			{
				// If met delimeter separated values, read one by one
				const unsigned Separately = 1;

				// If met delimeter separated values, read as vector
				const unsigned Continuously = (1 << 1);
			}

			namespace Check {}
		}

		namespace Vector
		{
			namespace Parse
			{
				// Read vector's size first (automatically check for items count)
				const unsigned Size = 1;

				// Read vector's elements line by line
				const unsigned ItemsLineByLine = (1 << 1);

				// Read vector's elements at one as a delimeter-separated string
				const unsigned ItemsAtOnce = (1 << 2);
			}

			namespace Check
			{
				// All loaded vectors should have the same length
				const unsigned LengthEquality = (1 << 3);
			}
		}

		namespace Matrix
		{
			/*  
				-- Work with only 2D matrix
				-- Matrix should always has dimentions being put before the data
				-- The number of rows and columns are checked after full read
			*/

			namespace Parse
			{
				// Read dimentions: each value on separate line {val_1 \n val_2}
				const unsigned DimentionsLineByLine = 1;

				// Read dimentions: both values on the same line {val_1 <delim> val_2 }
				const unsigned DimentionsAtOnce = (1 << 1);
			}

			namespace Check {}
		}
	}

	// Namespace Aliases
	namespace Flags_Arbitrary_Parse = Flags::Arbitrary::Parse;
	namespace Flags_Arbitrary_Check = Flags::Arbitrary::Check;
	namespace Flags_Vector_Parse = Flags::Vector::Parse;
	namespace Flags_Vector_Check = Flags::Vector::Check;
	namespace Flags_Matrix_Parse = Flags::Matrix::Parse;
	namespace Flags_Matrix_Check = Flags::Matrix::Check;

	namespace State
	{
		const unsigned Idle = 1; // ok
		const unsigned FileAccessError = (1 << 1); // bad
		const unsigned FileReadFormatError = (1 << 2); // fail
		const unsigned FileReadFailure = (1 << 3); // bad/ios::bad
		const unsigned VectorItemsWrongNumber = (1 << 4); // fail
		const unsigned VectorsHaveDifferentLength = (1 << 4); // fail
		const unsigned WrongReadConfiguration = (1 << 5); // fail
		const unsigned VectorInvalidIndex = (1 << 5); // bad
	}

	// Restricting the usage of Loader class
	template<class U>
	struct EnableForArithmeticType {
		typedef typename std::enable_if<std::is_arithmetic<U>::value>::type type;
	};

	template<typename U, typename Enable = EnableForArithmeticType<U>::type>
	class Loader
	{
	private: 

		// Name of the input data file
		std::string _file;

		// Default delimeter
		char _delimeter{ ',' };

		// Reader state
		unsigned _state{ State::Idle };

		// Service variables
		size_t sizeRow{ 0 }, sizeCol{ 0 };

		// Storage for read data
		std::vector<std::vector<U>> _storageVector;
		std::vector<U> _storageMatrix;

		void split(std::fstream& stream, std::vector<U>& storage, size_t lines_to_read)
		{
			std::string line;
			std::stringstream sstream;
			std::stringstream sconvert;
			std::string item;
			U num;

			while (!stream.eof() && lines_to_read--)
			{
				// Read vector's items
				std::getline(stream, line);

				if (!stream.eof() && stream.rdstate() != std::ios::goodbit)
				{
					if (stream.fail()) _state = State::FileReadFormatError;
					if (stream.bad()) _state = State::FileReadFailure;
					break;
				}
				
				sstream.str(line);				
				while (std::getline(sstream, item, _delimeter)) {
					sconvert.str(item);
					sconvert >> num;
					sconvert.clear();
					storage.push_back(num);
				}

				sstream.clear();
			}
		}

	public:

		explicit Loader(std::string file) : _file(file) {};

		// Settings: change delimeter character (',' by default)
		inline void delimeter(char delimeter) { _delimeter = delimeter; }

		// Service: get current state
		inline unsigned state() const { return _state; }

		// Service: state check aliases
		inline bool is_ok() const { return _state == State::Idle; };
		inline bool is_bad() const {
			return
				_state == State::FileAccessError ||
				_state == State::FileReadFailure ||
				_state == State::VectorInvalidIndex;
		};
		inline bool is_fail() const { return !is_ok() && !is_bad(); };

		// Service: reset internal state
		void refresh()
		{
			_state = State::Idle;
			_storageVector.clear();
		}

		// Service: get vector length
		size_t arg_vector_len(size_t idx)
		{
			if (idx >= _storageVector.size())
			{
				_state = State::VectorInvalidIndex;
				return std::numeric_limits<size_t>::max();
			}

			return _storageVector[idx].size();
		}

		// Return i-th vector
		std::vector<U> arg_vector(size_t idx)
		{
			if (idx >= _storageVector.size())
				return std::vector<U>{};

			return _storageVector[idx];
		};

		// Fill in given vector storage
		void arg_vector(size_t idx, std::vector<U>& storage)
		{
			if (idx >= _storageVector.size())
				return;

			storage.clear();
			storage.reserve(_storageVector[idx].size());
			std::copy(_storageVector[idx].begin(), _storageVector[idx].end(), std::back_inserter(storage));
		};
		
		// Fill in given array storage
		void arg_vector(size_t idx, U* storage)
		{
			if (!storage || idx >= _storageVector.size())
				return;

			for (auto v : _storageVector[idx])
				*storage++ = U(v);
		}

		// Return flattened matrix
		std::vector<U> arg_matrix()
		{
			return _storageMatrix;
		}

		// Fill in given array storage
		void arg_matrix(U* storage)
		{
			if (!storage)
				return;

			for (auto v : _storageMatrix)
				*storage++ = U(v);
		}

		void arg_matrix_dims(int& row, int& col)
		{
			row = sizeRow;
			col = sizeCol;
		}
		
		// Parse input file without data specification
		void read_arbitrary(unsigned check_opts = Flags::NullOpts, 
			unsigned parse_opts = Flags_Arbitrary_Parse::Separately, Kind type = Kind::Arbitrary)
		{}

		// Parse input file for vector-based data
		void read_vector(unsigned check_opts = Flags::NullOpts, 
			unsigned parse_opts = Flags_Vector_Parse::ItemsAtOnce, Kind type = Kind::Vector)
		{
			size_t lenNextVector{ 0 }, lenAllVectors{ 0 };
			std::string line, line1, line2;
			std::fstream fstream;
			std::istringstream sstream;

			fstream.open(_file, std::ios_base::in);

			if (!fstream.is_open())
			{
				_state = State::FileAccessError;
				return;
			}

			_storageVector.clear();

			// Validate vector length if appropriate option is given
			bool checkVecLength = parse_opts & Flags_Vector_Parse::Size;
			// For each vector given check if items number remains the same
			bool checkVecEquality = check_opts & Flags_Vector_Check::LengthEquality;
			// If no Size is given and parse option is set to ItemsLineByLine,
			// we are to read in one vector storage until the end of the file
			bool readUntilEOF = !checkVecLength && (parse_opts & Flags_Vector_Parse::ItemsLineByLine);

			while (fstream.good())
			{
				std::vector<U> storageItem;

				// Read vector's length
				if (checkVecLength && std::getline(fstream, line))
				{
					sstream.str(line);
					sstream >> lenNextVector;
					sstream.clear();

					storageItem.reserve(lenNextVector);
				}
				
				size_t line_to_read = { 0 };
				// How many lines we need to read for a vector to be filled?
				//  -- if "ReadUntilEOF" flag is set, read until the EOF (lines to read = MAX(size_t))
				//  -- if "ItemsAtOnce" flag is set, read one line no matter if we are given size or not
				//  -- last case: flag "ItemsLineByLine" is set and "Size" flag is given, read "LenNextVector" lines
				line_to_read =
					readUntilEOF ? std::numeric_limits<std::size_t>::max() :
					parse_opts & Flags_Vector_Parse::ItemsAtOnce ? 1 :
					checkVecLength && (parse_opts & Flags_Vector_Parse::ItemsLineByLine) ? lenNextVector : 0;

				// Above were considered all the possible cases, so this condition will never
				// be executed, unless new conditions appear
				if (line_to_read == 0)
				{
					_state = State::WrongReadConfiguration;
					break;
				}

				// Load vector's elements
				split(fstream, storageItem, line_to_read);

				// Check if vectors are equal
				if (checkVecEquality)
				{
					if (lenAllVectors == 0)
						lenAllVectors = storageItem.size();
					else if (lenAllVectors != storageItem.size())
					{
						_state = State::VectorsHaveDifferentLength;
						break;
					}
				}

				// Check if given length for vector items is valid
				if (checkVecLength && lenNextVector != storageItem.size())
				{
					_state = State::VectorItemsWrongNumber;
					break;
				}

				// Add vector to the storage
				_storageVector.push_back(storageItem);
			}

			fstream.close();
		}

		// Parse input file for matrix-based data
		void read_matrix(unsigned check_opts = Flags::NullOpts, 
			unsigned parse_opts = Flags_Matrix_Parse::DimentionsAtOnce, Kind type = Kind::Matrix)
		{
			std::string line, line1, line2;
			std::fstream fstream;
			std::stringstream sstream;

			std::vector<U> storageItem;

			fstream.open(_file, std::ios_base::in);

			if (!fstream.is_open())
			{
				_state = State::FileAccessError;
				return;
			}

			_storageMatrix.clear();

			if (parse_opts & Flags_Matrix_Parse::DimentionsLineByLine)
			{
				std::getline(fstream, line1);
				std::getline(fstream, line2);

				// Row
				sstream.str(line1);
				sstream >> sizeRow;
				sstream.clear();

				// Col
				sstream.str(line2);
				sstream >> sizeCol;
				sstream.clear();
			}
			else if (parse_opts & Flags_Matrix_Parse::DimentionsAtOnce)
			{
				std::getline(fstream, line);
				sstream.str(line);
				sstream >> sizeRow >> sizeCol;
				sstream.clear();
			}
			else
			{
				_state = State::WrongReadConfiguration;
				return;
			}

			// Load matrix data
			_storageMatrix.reserve(sizeRow * sizeCol);
			storageItem.reserve(sizeCol);

			auto cnt = sizeRow;
			while (cnt--)
			{
				// Load vector's elements
				storageItem.clear();
				split(fstream, storageItem, 1);

				std::move(storageItem.begin(), storageItem.end(), std::back_inserter(_storageMatrix));
			}
			
			// Check matrix's elements length
			if (_storageMatrix.size() != sizeRow * sizeCol)
			{
				_state = State::VectorItemsWrongNumber;
				return;
			}

			fstream.close();
		}
	};
}
